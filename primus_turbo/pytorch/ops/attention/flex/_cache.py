###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Classification / detection caches (perf: avoid re-probing on reuse).

Probing a ``block_mask.mask_mod`` (up to 512x512) and re-running the ALiBi /
soft-cap ``score_mod`` detectors on *every* call is a fixed ~1-3 ms per-call cost
that dominates small/medium shapes. In real use the same ``block_mask`` /
``score_mod`` object is reused across layers and steps, so we memoise the
classification / detection *by object identity* (``weakref.WeakKeyDictionary`` so
entries vanish when the mask/score_mod is garbage-collected -- no leak, no
lifetime surprises). This is a pure speedup: a different object re-probes, and a
cache entry is exactly what the uncached path would have returned (identical
result / error semantics). A sentinel distinguishes a cached ``None`` result from
a miss; objects that cannot be weakly referenced simply skip the cache.
"""

import weakref
from typing import Any, Tuple

#
# Probing a ``block_mask.mask_mod`` (up to 512x512) and re-running the ALiBi /
# soft-cap ``score_mod`` detectors on *every* call is a fixed ~1-3 ms per-call cost
# that dominates small/medium shapes. In real use the same ``block_mask`` /
# ``score_mod`` object is reused across layers and steps, so we memoise the
# classification / detection *by object identity* (``weakref.WeakKeyDictionary`` so
# entries vanish when the mask/score_mod is garbage-collected -- no leak, no
# lifetime surprises). This is a pure speedup: a different object re-probes, and a
# cache entry is exactly what the uncached path would have returned (identical
# result / error semantics). A sentinel distinguishes a cached ``None`` result from
# a miss; objects that cannot be weakly referenced simply skip the cache.
_CACHE_MISS = object()


# block_mask obj -> {(B, H, q_len, kv_len): cfg dict}
_CLASSIFY_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


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
    _ALIBI_CACHE.clear()
    _SOFTCAP_CACHE.clear()
