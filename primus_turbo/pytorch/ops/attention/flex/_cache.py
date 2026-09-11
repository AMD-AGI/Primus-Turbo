###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Classification / detection caches (perf: avoid re-probing on reuse).

Probing a ``block_mask.mask_mod`` (up to 512x512) and re-running the ALiBi /
soft-cap ``score_mod`` detectors on *every* call is a per-call cost that dominates
small and medium shapes. Measured host-side on MI355 (gfx950): 0.6-1.8 ms for the
causal and sliding-window shapes, but 4.6 ms at S=1024 rising to 19 ms at S=8192
for document packing, whose recognition verifies the reconstruction exactly over
the whole sequence instead of only the probed corner -- so the cost is neither
fixed nor small, and at S=8192 it is ~10x the attention kernel it precedes. A warm
hit is ~0.3 us, i.e. free. In real use the same ``mask_mod`` /
``score_mod`` object is reused across layers and steps, so we memoise the
classification / detection *by object identity* (with a content-fingerprint
fallback for mask_mods that a factory rebuilds each step) (``weakref.WeakKeyDictionary`` so
entries vanish when the mask/score_mod is garbage-collected -- no leak, no
lifetime surprises). This is a pure speedup: a different object re-probes, and a
cache entry is exactly what the uncached path would have returned (identical
result / error semantics). A sentinel distinguishes a cached ``None`` result from
a miss; objects that cannot be weakly referenced simply skip the cache.
"""

import weakref
from typing import Any, Tuple

_CACHE_MISS = object()


# mask_mod obj -> {(B, H, q_len, kv_len): cfg dict}
#
# Keyed on the ``mask_mod``, NOT on the ``BlockMask`` that carries it. The
# classifier reads exactly two things -- ``block_mask.mask_mod`` and the shape --
# so the BlockMask wrapper contributes nothing to the answer, and keying on it
# made every hit conditional on the wrapper surviving. Real training rebuilds the
# BlockMask every step (``create_block_mask`` per forward) while passing the same
# module-level ``mask_mod``, so the old key missed every single time: measured
# end to end, 34-38 s of cold classify + BlockMask build + compile per unique
# mask/shape, paid once per step instead of once per run.
_CLASSIFY_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


# Fallback for mask_mods that are *rebuilt* each step rather than reused -- a
# factory like ``def win(w): def m(...): ... ; return m`` hands out a fresh
# function object every call, so identity misses even though the behaviour is
# identical. Key such functions by content instead: the code object (strongly
# referenced, so no id-reuse hazard), the module/qualname it was defined in, the
# default arguments, and the captured closure values.
#
# Deliberately conservative -- ``_fn_fingerprint`` returns None unless every
# captured value is a small immutable. A closure over a tensor would otherwise be
# pinned alive by the cache key for the life of the process, which is a leak, and
# a closure over a mutable (a list someone appends to) would key equal while
# behaving differently, which is a correctness bug. Both fall back to the
# identity cache above, i.e. to today's behaviour.
_FINGERPRINT_CACHE: dict = {}

# A bound, not a tuning knob: a training run has a handful of distinct masks, so
# anything past this means fingerprints are being generated per step and the
# cache is not working. Drop it all rather than grow without limit.
_FINGERPRINT_CACHE_MAX = 256

_FP_SCALARS = (int, float, bool, str, bytes, type(None))


def _fp_value_ok(v: Any) -> bool:
    """True when ``v`` is a small immutable safe to embed in a cache key."""
    if isinstance(v, _FP_SCALARS):
        return True
    if isinstance(v, (tuple, frozenset)):
        return all(_fp_value_ok(x) for x in v)
    return False


def _fn_fingerprint(fn: Any) -> Any:
    """Content key for ``fn``, or ``None`` when one cannot be formed safely.

    ``None`` is not a failure -- it means "use identity", which is what the layer
    did before this cache existed.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    try:
        cells = tuple(c.cell_contents for c in (fn.__closure__ or ()))
    except ValueError:
        # An empty cell: the closure is still being constructed. Not fingerprintable.
        return None
    defaults = getattr(fn, "__defaults__", None) or ()
    if not all(_fp_value_ok(v) for v in cells) or not all(_fp_value_ok(v) for v in defaults):
        return None
    return (code, getattr(fn, "__module__", None), getattr(fn, "__qualname__", None), defaults, cells)


def _fingerprint_get(fn: Any, key: Tuple) -> Any:
    """Return the content-cached value for ``(fn, key)`` or ``_CACHE_MISS``."""
    fp = _fn_fingerprint(fn)
    if fp is None:
        return _CACHE_MISS
    return _FINGERPRINT_CACHE.get((fp, key), _CACHE_MISS)


def _fingerprint_put(fn: Any, key: Tuple, value: Any) -> None:
    """Memoise ``value`` against ``fn``'s content key; no-op when unfingerprintable."""
    fp = _fn_fingerprint(fn)
    if fp is None:
        return
    if len(_FINGERPRINT_CACHE) >= _FINGERPRINT_CACHE_MAX:
        _FINGERPRINT_CACHE.clear()
    _FINGERPRINT_CACHE[(fp, key)] = value


# score_mod obj -> {(B, Hq, q_len, kv_len): slopes tensor or None}
_ALIBI_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


# score_mod obj -> {(B, Hq, q_len, kv_len): cap float or None}
_SOFTCAP_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _cache_get(cache: "weakref.WeakKeyDictionary", obj: Any, key: Tuple) -> Any:
    """Return the cached value for ``(obj, key)`` or ``_CACHE_MISS``.

    Tolerates objects that cannot be weakly referenced (returns a miss) so caching
    never changes behaviour -- only speed.
    """
    try:
        inner = cache.get(obj)
    except TypeError:
        return _CACHE_MISS
    if inner is None:
        return _CACHE_MISS
    return inner.get(key, _CACHE_MISS)


def _cache_put(cache: "weakref.WeakKeyDictionary", obj: Any, key: Tuple, value: Any) -> None:
    """Memoise ``value`` for ``(obj, key)``; silently skip non-weak-referenceable objects."""
    try:
        inner = cache.get(obj)
        if inner is None:
            inner = {}
            cache[obj] = inner
    except TypeError:
        return
    inner[key] = value


def clear_classification_cache() -> None:
    """Clear all mask-classification / score_mod-detection caches.

    Rarely needed (entries are keyed by object identity and drop automatically when
    the mask / score_mod is collected), but handy for tests / benchmarks that want a
    cold-cache measurement or full determinism.
    """
    _CLASSIFY_CACHE.clear()
    _FINGERPRINT_CACHE.clear()
    _ALIBI_CACHE.clear()
    _SOFTCAP_CACHE.clear()
