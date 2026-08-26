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
    # An object that cannot be weak-referenced must still classify correctly, just
    # without caching (graceful fallback, no crash, behaviour preserved).
    obj = _NoWeakrefBlockMask(lambda b, h, q, kv: q >= kv)
    cfg1 = _classify_block_mask(obj, B=1, H=1, q_len=64, kv_len=64)
    cfg2 = _classify_block_mask(obj, B=1, H=1, q_len=64, kv_len=64)
    assert cfg1 == cfg2 == {"kind": "causal", "causal": True, "window_size": (-1, -1)}
    assert cfg1 is not cfg2  # not cached (cannot weakref) -> recomputed


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
