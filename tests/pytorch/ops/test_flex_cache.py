###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``flex/_cache.py``: identity-keyed classification / detection caching (perf; behaviour unchanged)."""

import torch

from primus_turbo.pytorch.ops.attention.flex import _mask_classify as flex_mask_classify
from primus_turbo.pytorch.ops.attention.flex import _score_mod as flex_score_mod
from primus_turbo.pytorch.ops.attention.flex._cache import clear_classification_cache
from primus_turbo.pytorch.ops.attention.flex._mask_classify import (
    _classify_block_mask,
    _classify_block_mask_uncached,
)
from primus_turbo.pytorch.ops.attention.flex._score_mod import (
    _cached_detect_alibi_slopes,
    _cached_detect_softcap,
    _detect_alibi_slopes,
)
from primus_turbo.pytorch.ops.attention.flex_attention_interface import flex_attention

from .flex_test_utils import _alibi_score_mod, _DummyBlockMask, _make_qkv, _softcap_score_mod


class _NoWeakrefBlockMask:
    """A block_mask that cannot be weakly referenced (no __weakref__ slot)."""

    __slots__ = ("mask_mod",)

    def __init__(self, mask_mod):
        self.mask_mod = mask_mod


class _NoWeakrefMaskMod:
    """A *callable* mask_mod that cannot be weakly referenced, and has no ``__code__``.

    The classification cache is keyed on the mask_mod, so this -- not the BlockMask
    wrapper -- is what has to degrade gracefully: neither the identity cache
    (needs a weakref) nor the fingerprint cache (needs ``__code__``) can hold it.
    """

    __slots__ = ()

    def __call__(self, b, h, q_idx, kv_idx):
        return q_idx >= kv_idx


# ---- classify cache -------------------------------------------------------


def test_classify_cache_returns_same_object_on_reuse():
    bm = _DummyBlockMask(lambda b, h, q, kv: q >= kv)
    cfg1 = _classify_block_mask(bm, B=1, H=1, q_len=64, kv_len=64)
    cfg2 = _classify_block_mask(bm, B=1, H=1, q_len=64, kv_len=64)
    assert cfg1 is cfg2  # cache hit returns the identical object


def test_classify_cache_distinct_objects_recompute():
    bm1 = _DummyBlockMask(lambda b, h, q, kv: q >= kv)
    bm2 = _DummyBlockMask(lambda b, h, q, kv: q >= kv)
    cfg1 = _classify_block_mask(bm1, B=1, H=1, q_len=64, kv_len=64)
    cfg2 = _classify_block_mask(bm2, B=1, H=1, q_len=64, kv_len=64)
    assert cfg1 is not cfg2  # different object -> re-probed
    assert cfg1 == cfg2  # ... but same classification content


def test_classify_cache_distinct_shape_recompute():
    bm = _DummyBlockMask(lambda b, h, q, kv: q >= kv)
    cfg1 = _classify_block_mask(bm, B=1, H=1, q_len=64, kv_len=64)
    cfg2 = _classify_block_mask(bm, B=1, H=1, q_len=128, kv_len=128)
    assert cfg1 is not cfg2  # shape is part of the key


def test_classify_cache_behaviour_matches_uncached():
    for mask_mod in (
        lambda b, h, q, kv: True,
        lambda b, h, q, kv: q >= kv,
        lambda b, h, q, kv: (q >= kv) & ((q - kv) <= 5),
    ):
        bm = _DummyBlockMask(mask_mod)
        cached = _classify_block_mask(bm, B=1, H=1, q_len=32, kv_len=32)
        uncached = _classify_block_mask_uncached(bm, B=1, H=1, q_len=32, kv_len=32)
        assert cached == uncached


def test_classify_cache_none_short_circuits_full():
    assert _classify_block_mask(None, B=2, H=4, q_len=128, kv_len=128)["kind"] == "full"


def test_classify_cache_skips_non_weakreferenceable():
    # A mask_mod that can be neither weak-referenced nor fingerprinted must still
    # classify correctly, just without caching: graceful fallback, no crash,
    # behaviour preserved.
    obj = _NoWeakrefBlockMask(_NoWeakrefMaskMod())
    cfg1 = _classify_block_mask(obj, B=1, H=1, q_len=64, kv_len=64)
    cfg2 = _classify_block_mask(obj, B=1, H=1, q_len=64, kv_len=64)
    assert cfg1 == cfg2 == {"kind": "causal", "causal": True, "window_size": (-1, -1)}
    assert cfg1 is not cfg2  # not cached -> recomputed


def test_a_non_weakreferenceable_block_mask_no_longer_defeats_the_cache():
    # The wrapper's weakref-ability is irrelevant now that the key is the mask_mod.
    # This is the whole point of the re-key: the BlockMask is rebuilt every step and
    # must not be what the cache depends on.
    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    cfg1 = _classify_block_mask(_NoWeakrefBlockMask(mask_mod), B=1, H=1, q_len=64, kv_len=64)
    cfg2 = _classify_block_mask(_NoWeakrefBlockMask(mask_mod), B=1, H=1, q_len=64, kv_len=64)
    assert cfg1 is cfg2


def test_classify_cache_used_by_flex_attention(capture_backend, monkeypatch):
    # End-to-end: two calls with the SAME block_mask probe the mask only once.
    calls = {"n": 0}
    orig = flex_mask_classify._classify_block_mask_uncached

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(flex_mask_classify, "_classify_block_mask_uncached", spy)
    clear_classification_cache()
    q, k, v = _make_qkv(Hq=4, S=16, D=16)
    bm = _DummyBlockMask(lambda b, h, qi, ki: qi >= ki)
    flex_attention(q, k, v, block_mask=bm)
    flex_attention(q, k, v, block_mask=bm)
    assert calls["n"] == 1  # second call hit the cache


# ---- score_mod (alibi / softcap) detection cache --------------------------


def test_cached_detect_alibi_same_tensor_on_reuse():
    sm = _alibi_score_mod(8)
    s1 = _cached_detect_alibi_slopes(sm, B=1, Hq=8, q_len=64, kv_len=64)
    s2 = _cached_detect_alibi_slopes(sm, B=1, Hq=8, q_len=64, kv_len=64)
    assert s1 is s2  # cache hit returns the identical tensor


def test_cached_detect_alibi_distinct_objects_recompute():
    sm1 = _alibi_score_mod(8)
    sm2 = _alibi_score_mod(8)
    s1 = _cached_detect_alibi_slopes(sm1, B=1, Hq=8, q_len=64, kv_len=64)
    s2 = _cached_detect_alibi_slopes(sm2, B=1, Hq=8, q_len=64, kv_len=64)
    assert s1 is not s2
    assert torch.equal(s1, s2)


def test_cached_detect_alibi_behaviour_matches_uncached():
    sm = _alibi_score_mod(8)
    cached = _cached_detect_alibi_slopes(sm, B=1, Hq=8, q_len=64, kv_len=64)
    uncached = _detect_alibi_slopes(sm, B=1, Hq=8, q_len=64, kv_len=64)
    assert torch.equal(cached, uncached)


def test_cached_detect_alibi_caches_none():
    def not_alibi(score, b, h, q_idx, kv_idx):
        return score + 0.1 * score

    r1 = _cached_detect_alibi_slopes(not_alibi, B=1, Hq=8, q_len=64, kv_len=64)
    r2 = _cached_detect_alibi_slopes(not_alibi, B=1, Hq=8, q_len=64, kv_len=64)
    assert r1 is None and r2 is None  # cached None is still None (behaviour unchanged)


def test_cached_detect_softcap_hits_cache(monkeypatch):
    calls = {"n": 0}
    orig = flex_score_mod._detect_softcap

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(flex_score_mod, "_detect_softcap", spy)
    clear_classification_cache()
    sm = _softcap_score_mod(30.0)
    c1 = _cached_detect_softcap(sm, B=1, Hq=8, q_len=64, kv_len=64)
    c2 = _cached_detect_softcap(sm, B=1, Hq=8, q_len=64, kv_len=64)
    assert calls["n"] == 1  # second call hit the cache
    assert c1 == c2 and c1 is not None
    assert abs(c1 - 30.0) < 1e-2 * 30.0  # behaviour unchanged


def test_cached_detect_softcap_distinct_objects_recompute(monkeypatch):
    calls = {"n": 0}
    orig = flex_score_mod._detect_softcap

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(flex_score_mod, "_detect_softcap", spy)
    clear_classification_cache()
    _cached_detect_softcap(_softcap_score_mod(30.0), B=1, Hq=8, q_len=64, kv_len=64)
    _cached_detect_softcap(_softcap_score_mod(30.0), B=1, Hq=8, q_len=64, kv_len=64)
    assert calls["n"] == 2  # different score_mod objects -> recomputed


def test_clear_classification_cache_forces_recompute():
    bm = _DummyBlockMask(lambda b, h, q, kv: q >= kv)
    cfg1 = _classify_block_mask(bm, B=1, H=1, q_len=64, kv_len=64)
    clear_classification_cache()
    cfg2 = _classify_block_mask(bm, B=1, H=1, q_len=64, kv_len=64)
    assert cfg1 is not cfg2  # cache cleared -> recomputed (new object)
    assert cfg1 == cfg2


# --- T23: the cache has to survive a rebuilt BlockMask ----------------------
# Real training calls create_block_mask every forward. Keying the classification
# on the BlockMask wrapper meant a guaranteed miss every step: measured end to
# end, 34-38 s of cold classify + build + compile per unique mask/shape.


def _win(w):
    """A factory, so each call hands out a *fresh* function object."""

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= w)

    return mask_mod


def _count_probes(monkeypatch):
    """Count how many times the uncached classifier actually runs."""
    from primus_turbo.pytorch.ops.attention.flex import _mask_classify as mc

    calls = []
    real = mc._classify_block_mask_uncached

    def counting(block_mask, **kw):
        calls.append(kw)
        return real(block_mask, **kw)

    monkeypatch.setattr(mc, "_classify_block_mask_uncached", counting, raising=True)
    return calls


def test_rebuilt_block_mask_around_the_same_mask_mod_still_hits(monkeypatch):
    calls = _count_probes(monkeypatch)
    mask_mod = _win(64)  # built once, reused -- the module-level-function case

    first = _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=256, kv_len=256)
    # A brand new wrapper object every step, exactly like create_block_mask-per-forward.
    for _ in range(4):
        again = _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=256, kv_len=256)
        assert again == first

    assert len(calls) == 1, "rebuilding the BlockMask must not re-probe"


def test_a_rebuilt_mask_mod_hits_via_the_content_fingerprint(monkeypatch):
    calls = _count_probes(monkeypatch)
    # Fresh function object each step, same captured window: same behaviour.
    first = _classify_block_mask(_DummyBlockMask(_win(64)), B=1, H=1, q_len=256, kv_len=256)
    for _ in range(4):
        assert _classify_block_mask(_DummyBlockMask(_win(64)), B=1, H=1, q_len=256, kv_len=256) == first
    assert len(calls) == 1


def test_a_different_captured_value_is_a_different_key(monkeypatch):
    # The fingerprint must not collapse w=64 and w=128 onto one entry -- that would
    # silently train the wrong mask, which is the whole class of bug this PR hunts.
    calls = _count_probes(monkeypatch)
    a = _classify_block_mask(_DummyBlockMask(_win(64)), B=1, H=1, q_len=256, kv_len=256)
    b = _classify_block_mask(_DummyBlockMask(_win(128)), B=1, H=1, q_len=256, kv_len=256)
    assert a["window_size"] == (64, 0)
    assert b["window_size"] == (128, 0)
    assert len(calls) == 2


def test_a_different_shape_is_a_different_key(monkeypatch):
    calls = _count_probes(monkeypatch)
    mask_mod = _win(64)
    _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=256, kv_len=256)
    _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=512, kv_len=512)
    assert len(calls) == 2


def test_a_closure_over_a_mutable_is_not_fingerprinted():
    # A list someone appends to would key equal while behaving differently. Refusing
    # to fingerprint it falls back to identity, i.e. to the old behaviour -- correct,
    # just not cached across rebuilds.
    from primus_turbo.pytorch.ops.attention.flex._cache import _fn_fingerprint

    bounds = [128]

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx + bounds[0]

    assert _fn_fingerprint(mask_mod) is None


def test_a_closure_over_a_tensor_is_not_fingerprinted():
    # Pinning a tensor alive inside a process-lifetime cache key is a leak.
    from primus_turbo.pytorch.ops.attention.flex._cache import _fn_fingerprint

    captured = torch.zeros(4)

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx + int(captured.numel())

    assert _fn_fingerprint(mask_mod) is None


def test_fingerprint_cache_is_bounded():
    from primus_turbo.pytorch.ops.attention.flex import _cache as cache_mod

    cache_mod.clear_classification_cache()
    for i in range(cache_mod._FINGERPRINT_CACHE_MAX + 5):
        cache_mod._fingerprint_put(_win(i), (1, 1, 256, 256), {"kind": "sliding_window_causal"})
    assert len(cache_mod._FINGERPRINT_CACHE) <= cache_mod._FINGERPRINT_CACHE_MAX
