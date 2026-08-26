###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``flex/_routing.py``: backend selection, the override registry and backend feature probing."""

import pytest
import torch

from primus_turbo.pytorch.ops.attention.flex import _routing as flex_routing
from primus_turbo.pytorch.ops.attention.flex._routing import (
    _backend_accepts,
    choose_backend,
    clear_backend_overrides,
    register_backend_override,
)
from primus_turbo.pytorch.ops.attention.flex_attention_interface import flex_attention_varlen

from .flex_test_utils import _CAUSAL_CFG, _cu_from_seqlens, _make_thd


def test_choose_backend_defaults_to_turbo():
    # No overrides registered -> every recognised variant stays on Turbo.
    got = choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)
    assert got == "turbo"


def test_choose_backend_default_turbo_across_kinds():
    for cfg in (
        {"kind": "full", "causal": False, "window_size": (-1, -1)},
        {"kind": "causal", "causal": True, "window_size": (-1, -1)},
        {"kind": "sliding_window_causal", "causal": True, "window_size": (128, 0)},
    ):
        assert choose_backend(cfg, shape=(2, 4, 256, 64), dtype=torch.float16, has_alibi=True) == "turbo"


def test_register_backend_override_routes_custom():
    # An override matching this mask kind must reroute it to the custom hook.
    register_backend_override(lambda ctx: ctx["kind"] == "causal", "custom")
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False) == "custom"
    )
    # A non-matching kind is unaffected.
    full_cfg = {"kind": "full", "causal": False, "window_size": (-1, -1)}
    assert choose_backend(full_cfg, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False) == "turbo"


def test_clear_backend_overrides_restores_turbo():
    register_backend_override(lambda ctx: True, "custom")
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False) == "custom"
    )
    clear_backend_overrides()
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False) == "turbo"
    )


def test_backend_override_first_match_wins():
    register_backend_override(lambda ctx: ctx["kind"] == "causal", "custom")
    register_backend_override(lambda ctx: ctx["kind"] == "causal", "turbo")
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False) == "custom"
    )


def test_backend_override_matches_on_shape_and_softcap():
    # Matchers can key off any routing-context field, e.g. a large head dim or softcap.
    register_backend_override(lambda ctx: ctx["shape"][-1] > 128, "custom")
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 256), dtype=torch.bfloat16, has_alibi=False) == "custom"
    )
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False) == "turbo"
    )

    clear_backend_overrides()
    register_backend_override(lambda ctx: ctx["has_softcap"], "custom")
    assert (
        choose_backend(
            _CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False, has_softcap=True
        )
        == "custom"
    )


def test_choose_backend_ctx_exposes_expected_fields():
    seen = {}

    def matcher(ctx):
        seen.update(ctx)
        return False

    register_backend_override(matcher, "custom")
    got = choose_backend(
        {"kind": "sliding_window_causal", "causal": True, "window_size": (64, 0)},
        shape=(3, 5, 128, 64),
        dtype=torch.float16,
        has_alibi=True,
        has_softcap=True,
        has_dropout=True,
        has_sink=True,
        has_bias=True,
    )
    assert got == "turbo"  # matcher returned False -> default
    for key in (
        "kind",
        "causal",
        "window_size",
        "shape",
        "dtype",
        "has_alibi",
        "has_softcap",
        "has_dropout",
        "has_sink",
        "has_bias",
        "mask_cfg",
    ):
        assert key in seen
    assert seen["kind"] == "sliding_window_causal"
    assert seen["shape"] == (3, 5, 128, 64)
    assert seen["has_alibi"] is True
    assert seen["has_softcap"] is True
    assert seen["has_dropout"] is True
    assert seen["has_sink"] is True
    assert seen["has_bias"] is True


def test_choose_backend_has_dropout_sink_default_false():
    # Omitting the new flags keeps them false (backward compatible with old callers).
    seen = {}

    def matcher(ctx):
        seen.update(ctx)
        return False

    register_backend_override(matcher, "custom")
    choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)
    assert seen["has_dropout"] is False
    assert seen["has_sink"] is False
    assert seen["has_bias"] is False


def test_backend_override_matches_on_dropout_and_sink():
    register_backend_override(lambda ctx: ctx["has_dropout"], "custom")
    assert (
        choose_backend(
            _CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False, has_dropout=True
        )
        == "custom"
    )
    clear_backend_overrides()
    register_backend_override(lambda ctx: ctx["has_sink"], "custom")
    assert (
        choose_backend(
            _CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False, has_sink=True
        )
        == "custom"
    )


def test_register_backend_override_validates_backend():
    with pytest.raises(ValueError):
        register_backend_override(lambda ctx: True, "not_a_backend")


def test_register_backend_override_validates_matcher():
    with pytest.raises(TypeError):
        register_backend_override("not_callable", "custom")


def test_choose_backend_wraps_matcher_errors():
    def boom(ctx):
        raise KeyError("missing")

    register_backend_override(boom, "custom")
    with pytest.raises(RuntimeError):
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)


# These mirror the real ``flash_attn_varlen_func`` keyword set so that a call can
# actually go through them; the only difference between the two is the presence of
# ``sink``, which is exactly the build difference being probed.
def _fn_with_sink(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max_q,
    max_k,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    sink=None,
):
    return q


def _fn_without_sink(
    q,
    k,
    v,
    cu_q,
    cu_k,
    max_q,
    max_k,
    *,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
):
    return q


def _fn_var_kwargs(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs):
    return q


def test_backend_accepts_detects_present_parameter():
    assert _backend_accepts(_fn_with_sink, "sink") is True


def test_backend_accepts_detects_absent_parameter():
    assert _backend_accepts(_fn_without_sink, "sink") is False


def test_backend_accepts_treats_var_kwargs_as_permissive():
    # A backend declared as **kwargs could accept anything; refusing pre-emptively
    # would break perfectly good builds, so it is given the benefit of the doubt.
    assert _backend_accepts(_fn_var_kwargs, "sink") is True
    assert _backend_accepts(_fn_var_kwargs, "anything_at_all") is True


def test_backend_accepts_unintrospectable_callable_is_permissive(monkeypatch):
    # C-implemented callables can have no retrievable signature at all. Rejecting
    # them would make the compat layer unusable against a compiled backend, so the
    # probe has to fall back to "assume it works" rather than "assume it doesn't".
    # (Most builtins DO carry a signature these days, so the condition is forced
    # here rather than hoping to find a genuinely opaque one.)
    def boom(_fn):
        raise ValueError("no signature found for builtin")

    monkeypatch.setattr(flex_routing.inspect, "signature", boom)
    opaque = object()
    assert _backend_accepts(opaque, "sink") is True


def test_backend_accepts_is_cached_but_not_confused_by_id_reuse():
    # The cache is keyed on id(fn) for speed; it must also verify identity, because
    # CPython recycles the ids of collected objects. Two callables that happen to
    # land on the same id must not inherit each other's parameter set.
    assert _backend_accepts(_fn_with_sink, "sink") is True
    assert _backend_accepts(_fn_with_sink, "sink") is True  # second call hits the cache
    assert _backend_accepts(_fn_without_sink, "sink") is False

    def make():
        def tmp(q, k, v, cu_q, cu_k, max_q, max_k, *, sink=None):
            return q

        return tmp

    seen = set()
    for _ in range(50):
        fn = make()
        assert _backend_accepts(fn, "sink") is True
        seen.add(id(fn))
        del fn  # let the id be recycled by the next iteration
    # Whatever ids were recycled, every probe above still answered from the real
    # signature rather than a stale cache entry.
    assert len(seen) >= 1


def _install_varlen_backend(monkeypatch, fn):
    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_varlen_func",
        fn,
        raising=True,
    )


def test_varlen_sink_raises_not_implemented_on_a_backend_without_it(monkeypatch):
    _install_varlen_backend(monkeypatch, _fn_without_sink)
    H, D = 8, 128
    q = _make_thd(512, H, D)
    cu, max_s, _ = _cu_from_seqlens([128, 128, 256])
    sink = torch.zeros(H, dtype=torch.float32)
    with pytest.raises(NotImplementedError) as exc:
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, sink=sink)
    msg = str(exc.value)
    assert "sink" in msg
    # The message must point at the backend build, not look like a compat-layer bug,
    # and must offer the path that does work.
    assert "flash_attn_varlen_func" in msg
    assert "flex_attention_bshd" in msg


def test_varlen_without_a_sink_still_works_on_a_backend_without_it(monkeypatch):
    # The whole point of threading ``sink`` conditionally: an older backend must stay
    # usable for every call that does not ask for a sink.
    _install_varlen_backend(monkeypatch, _fn_without_sink)
    H, D = 8, 128
    q = _make_thd(512, H, D)
    cu, max_s, _ = _cu_from_seqlens([128, 128, 256])
    out = flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True)
    assert out.shape == (512, H, D)


def test_varlen_sink_is_threaded_when_the_backend_takes_it(monkeypatch):
    seen = {}

    def fn(q, k, v, cu_q, cu_k, max_q, max_k, *, sink=None, **kwargs):
        seen["sink"] = sink
        return q

    _install_varlen_backend(monkeypatch, fn)
    H, D = 8, 128
    q = _make_thd(512, H, D)
    cu, max_s, _ = _cu_from_seqlens([128, 128, 256])
    sink = torch.arange(H, dtype=torch.float32)
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, sink=sink)
    assert seen["sink"] is not None
    assert torch.equal(seen["sink"], sink)


def test_varlen_invalid_sink_is_rejected_before_the_backend_is_consulted(monkeypatch):
    # A malformed sink is the caller's error regardless of backend capability, so the
    # validator must fire first -- otherwise a build without the parameter would
    # report "your backend is too old" for what is really a bad argument.
    _install_varlen_backend(monkeypatch, _fn_without_sink)
    H, D = 8, 128
    q = _make_thd(512, H, D)
    cu, max_s, _ = _cu_from_seqlens([128, 128, 256])
    bad = torch.zeros(H, dtype=torch.bfloat16)  # must be fp32
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, sink=bad)
