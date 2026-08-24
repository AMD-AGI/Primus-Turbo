###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Pure-logic unit tests for the Turbo flex_attention dispatcher.

These exercise the mask classifier and the ALiBi score_mod detector without a
GPU: they only need torch on CPU.
"""

import math
import sys

import pytest
import torch

import primus_turbo.pytorch.ops.attention.flex_attention  # noqa: F401
from primus_turbo.pytorch.ops.attention.flex_attention import (
    _cached_detect_alibi_slopes,
    _cached_detect_softcap,
    _classify_block_mask,
    _classify_block_mask_uncached,
    _detect_alibi_slopes,
    _detect_document_causal_segments,
    _detect_softcap,
    _is_identity_score_mod,
    _locate_left_window,
    _normalise_explicit_softcap,
    _validate_and_adapt_bias,
    _validate_cu_seqlens,
    _validate_dropout_p,
    _validate_explicit_alibi_slopes,
    _validate_explicit_sink,
    _validate_max_seqlen,
    _validate_qkv_varlen,
    _validate_window_size,
    choose_backend,
    clear_backend_overrides,
    clear_classification_cache,
    flex_attention,
    flex_attention_bshd,
    flex_attention_varlen,
    register_backend_override,
)

# The package __init__ re-exports the *function* ``flex_attention`` under the same
# name as this submodule, so ``import ...flex_attention as fa_mod`` binds the
# function, not the module (plain-attribute shadowing). sys.modules always holds
# the real module, and works both against the installed package and against the
# CPU-only file-loading harness.
fa_mod = sys.modules["primus_turbo.pytorch.ops.attention.flex_attention"]


class _DummyBlockMask:
    def __init__(self, mask_mod):
        self.mask_mod = mask_mod


@pytest.fixture(autouse=True)
def _reset_backend_overrides():
    """The override registry and classification caches are module-global; keep
    tests independent (cold cache each test)."""
    clear_backend_overrides()
    clear_classification_cache()
    yield
    clear_backend_overrides()
    clear_classification_cache()


_CAUSAL_CFG = {"kind": "causal", "causal": True, "window_size": (-1, -1)}


# ---------------------------------------------------------------------------
# mask classification
# ---------------------------------------------------------------------------


def test_classify_full_mask():
    block_mask = _DummyBlockMask(lambda b, h, q, kv: True)
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=16, kv_len=16)
    assert cfg["kind"] == "full"
    assert cfg["causal"] is False
    assert cfg["window_size"] == (-1, -1)


def test_classify_causal_mask():
    block_mask = _DummyBlockMask(lambda b, h, q, kv: q >= kv)
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=16, kv_len=16)
    assert cfg["kind"] == "causal"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (-1, -1)


def test_classify_none_is_full():
    cfg = _classify_block_mask(None, B=2, H=4, q_len=128, kv_len=128)
    assert cfg["kind"] == "full"
    assert cfg["causal"] is False


@pytest.mark.parametrize("window", [1, 5, 63, 128, 256])
def test_classify_sliding_window_causal_mask(window):
    block_mask = _DummyBlockMask(lambda b, h, q, kv: (q >= kv) & ((q - kv) <= window))
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=512, kv_len=512)
    assert cfg["kind"] == "sliding_window_causal"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (window, 0)


def test_classify_sliding_window_python_and_scalar_fallback():
    # ``and`` only works on python scalars -> exercises the element-wise fallback.
    window = 5
    block_mask = _DummyBlockMask(lambda b, h, q, kv: (q >= kv) and ((q - kv) <= window))
    cfg = _classify_block_mask(block_mask, B=1, H=1, q_len=32, kv_len=32)
    assert cfg["kind"] == "sliding_window_causal"
    assert cfg["window_size"] == (window, 0)


def test_classify_random_mask_unsupported():
    block_mask = _DummyBlockMask(lambda b, h, q, kv: ((q + kv) % 2) == 0)
    with pytest.raises(NotImplementedError):
        _classify_block_mask(block_mask, B=1, H=1, q_len=16, kv_len=16)


def test_classify_bidirectional_band_unsupported():
    # Symmetric band around the diagonal (non-causal) -> not expressible.
    block_mask = _DummyBlockMask(lambda b, h, q, kv: (kv - q).__abs__() <= 2)
    with pytest.raises(NotImplementedError):
        _classify_block_mask(block_mask, B=1, H=1, q_len=32, kv_len=32)


def test_classify_head_dependent_unsupported():
    # Head 0 causal, other heads full -> depends on h, cannot map to one kernel call.
    def mask_mod(b, h, q, kv):
        return (q >= kv) | (h > 0)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=4, q_len=32, kv_len=32)


def test_classify_batch_dependent_unsupported():
    def mask_mod(b, h, q, kv):
        return (q >= kv) | (b > 0)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=4, H=1, q_len=32, kv_len=32)


# ---------------------------------------------------------------------------
# ALiBi score_mod detection
# ---------------------------------------------------------------------------


def _alibi_score_mod(num_heads):
    def score_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * float(h) / num_heads)
        return score + slope * (kv_idx - q_idx)

    return score_mod


def test_detect_alibi_recovers_slopes():
    H = 8
    slopes = _detect_alibi_slopes(_alibi_score_mod(H), B=1, Hq=H, q_len=64, kv_len=64)
    assert slopes is not None
    expected = [2.0 ** (-8.0 * h / H) for h in range(H)]
    for got, exp in zip(slopes.tolist(), expected):
        assert abs(got - exp) < 1e-3


def test_detect_alibi_none_for_scale_score():
    # Multiplicative modification is not an additive (kv-q) bias.
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + 0.1 * score

    assert _detect_alibi_slopes(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_alibi_none_for_constant_bias():
    # Constant additive bias (non-zero at delta 0) is not ALiBi.
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + 1.0

    assert _detect_alibi_slopes(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_alibi_none_for_position_dependent():
    # Depends on absolute position (q_idx*kv_idx), not on (kv - q).
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + 0.001 * (q_idx * kv_idx)

    assert _detect_alibi_slopes(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_alibi_zero_for_identity():
    # A no-op score_mod is representable with zero slopes.
    def score_mod(score, b, h, q_idx, kv_idx):
        return score

    slopes = _detect_alibi_slopes(score_mod, B=1, Hq=4, q_len=64, kv_len=64)
    assert slopes is not None
    assert max(abs(x) for x in slopes.tolist()) < 1e-6


def test_detect_alibi_none_for_batch_dependent():
    H = 8

    def score_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * float(h) / H)
        return score + (1.0 + float(b)) * slope * (kv_idx - q_idx)

    assert _detect_alibi_slopes(score_mod, B=2, Hq=H, q_len=64, kv_len=64) is None


# ---------------------------------------------------------------------------
# softcap (logits soft-cap) detection
# ---------------------------------------------------------------------------


def _softcap_score_mod(cap):
    def score_mod(score, b, h, q_idx, kv_idx):
        return cap * math.tanh(score / cap)

    return score_mod


@pytest.mark.parametrize("cap", [20.0, 30.0, 50.0])
def test_detect_softcap_recovers_cap(cap):
    got = _detect_softcap(_softcap_score_mod(cap), B=1, Hq=8, q_len=64, kv_len=64)
    assert got is not None
    assert abs(got - cap) < 1e-2 * cap


def test_detect_softcap_none_for_identity():
    def score_mod(score, b, h, q_idx, kv_idx):
        return score

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_linear_scale():
    def score_mod(score, b, h, q_idx, kv_idx):
        return 1.1 * score

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_constant_bias():
    def score_mod(score, b, h, q_idx, kv_idx):
        return score + 1.0

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_alibi():
    # ALiBi depends on (kv - q) -> not a pure (position-independent) soft-cap.
    def score_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * float(h) / 8)
        return score + slope * (kv_idx - q_idx)

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_hard_clamp():
    # A hard clamp saturates like a soft-cap but is not tanh-shaped -> rejected.
    cap = 30.0

    def score_mod(score, b, h, q_idx, kv_idx):
        return max(-cap, min(cap, score))

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_detect_softcap_none_for_alibi_softcap_combo():
    # cap*tanh((score + alibi)/cap) is position-dependent -> not a pure soft-cap.
    cap = 30.0

    def score_mod(score, b, h, q_idx, kv_idx):
        slope = 2.0 ** (-8.0 * float(h) / 8)
        return cap * math.tanh((score + slope * (kv_idx - q_idx)) / cap)

    assert _detect_softcap(score_mod, B=1, Hq=8, q_len=64, kv_len=64) is None


def test_softcap_not_misdetected_as_zero_alibi():
    # Regression guard: the ALiBi detector only probes score=0 and would read a
    # soft-cap as zero slopes; the soft-cap detector must catch it first so the
    # dispatcher never silently drops the cap.
    cap = 30.0
    sm = _softcap_score_mod(cap)
    slopes = _detect_alibi_slopes(sm, B=1, Hq=8, q_len=64, kv_len=64)
    # ALiBi misreads it as (near-)zero slopes ...
    assert slopes is not None and max(abs(x) for x in slopes.tolist()) < 1e-2
    # ... but the soft-cap detector recognises it, so callers route it away from Turbo.
    assert _detect_softcap(sm, B=1, Hq=8, q_len=64, kv_len=64) is not None


# ---------------------------------------------------------------------------
# choose_backend / backend-override registry (performance routing layer)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Explicit Turbo-extension args: alibi_slopes validation helper
# ---------------------------------------------------------------------------


def test_validate_explicit_alibi_slopes_ok():
    slopes = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
    out = _validate_explicit_alibi_slopes(slopes, hq=3, device=torch.device("cpu"))
    assert out.shape == (3,)
    assert out.dtype == torch.float32
    assert torch.allclose(out, slopes)


def test_validate_explicit_alibi_slopes_rejects_non_tensor():
    with pytest.raises(ValueError):
        _validate_explicit_alibi_slopes([1.0, 0.5, 0.25], hq=3, device=torch.device("cpu"))


def test_validate_explicit_alibi_slopes_rejects_2d():
    slopes = torch.zeros((2, 3), dtype=torch.float32)
    with pytest.raises(ValueError):
        _validate_explicit_alibi_slopes(slopes, hq=3, device=torch.device("cpu"))


def test_validate_explicit_alibi_slopes_rejects_wrong_length():
    slopes = torch.zeros(3, dtype=torch.float32)
    with pytest.raises(ValueError):
        _validate_explicit_alibi_slopes(slopes, hq=8, device=torch.device("cpu"))


def test_validate_explicit_alibi_slopes_rejects_non_fp32():
    slopes = torch.zeros(4, dtype=torch.float16)
    with pytest.raises(ValueError):
        _validate_explicit_alibi_slopes(slopes, hq=4, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Explicit Turbo-extension args: softcap normalisation helper
# ---------------------------------------------------------------------------


def test_normalise_explicit_softcap_none_and_zero_disable():
    assert _normalise_explicit_softcap(None) == 0.0
    assert _normalise_explicit_softcap(0) == 0.0
    assert _normalise_explicit_softcap(0.0) == 0.0


def test_normalise_explicit_softcap_positive_kept():
    assert _normalise_explicit_softcap(30.0) == 30.0


def test_normalise_explicit_softcap_negative_raises():
    with pytest.raises(ValueError):
        _normalise_explicit_softcap(-1.0)


def test_normalise_explicit_softcap_nan_raises():
    with pytest.raises(ValueError):
        _normalise_explicit_softcap(float("nan"))


# ---------------------------------------------------------------------------
# Explicit Turbo-extension args: identity score_mod probe (conflict check)
# ---------------------------------------------------------------------------


def test_is_identity_score_mod_true_for_identity():
    assert _is_identity_score_mod(lambda s, b, h, q, kv: s, B=1, Hq=4, q_len=16, kv_len=16)


def test_is_identity_score_mod_true_for_add_zero():
    assert _is_identity_score_mod(lambda s, b, h, q, kv: s + 0.0, B=2, Hq=4, q_len=16, kv_len=16)


def test_is_identity_score_mod_false_for_alibi():
    def sm(s, b, h, q, kv):
        return s + 0.5 * (kv - q)

    assert not _is_identity_score_mod(sm, B=1, Hq=4, q_len=16, kv_len=16)


def test_is_identity_score_mod_false_for_constant_bias():
    assert not _is_identity_score_mod(lambda s, b, h, q, kv: s + 1.0, B=1, Hq=4, q_len=16, kv_len=16)


# ---------------------------------------------------------------------------
# Explicit Turbo-extension args: end-to-end dispatch (backend mocked on CPU)
#
# These exercise flex_attention's parameter-passing path without a GPU by
# patching the lazily-imported flash_attn_func and capturing the kwargs it would
# receive. Only fp16/bf16 4D CPU tensors are needed; the real kernel never runs.
# ---------------------------------------------------------------------------


@pytest.fixture
def capture_backend(monkeypatch):
    captured = {}

    def fake_flash_attn_func(q, k, v, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs
        captured["q_shape"] = tuple(q.shape)
        if kwargs.get("return_lse"):
            b, s, h, d = q.shape
            return q.clone(), torch.zeros((b, h, s), dtype=torch.float32)
        return q.clone()

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func",
        fake_flash_attn_func,
        raising=True,
    )
    return captured


def _make_qkv(B=1, Hq=4, S=16, D=16, dtype=torch.float16):
    q = torch.randn(B, Hq, S, D, dtype=dtype)
    k = torch.randn(B, Hq, S, D, dtype=dtype)
    v = torch.randn(B, Hq, S, D, dtype=dtype)
    return q, k, v


def test_explicit_alibi_slopes_passed_through_and_bypasses_detection(capture_backend):
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    out = flex_attention(q, k, v, alibi_slopes=slopes.clone())
    assert out.shape == (1, H, 16, 16)
    passed = capture_backend["kwargs"]["alibi_slopes"]
    assert passed is not None
    assert torch.allclose(passed.cpu(), slopes)


def test_explicit_alibi_equivalent_to_autodetected_slopes(capture_backend):
    # The explicit path and the score_mod auto-detect path must thread the *same*
    # per-head slopes to the backend for an equivalent ALiBi definition.
    H = 8
    q, k, v = _make_qkv(Hq=H, S=32)
    slopes = torch.tensor([2.0 ** (-8.0 * h / H) for h in range(H)], dtype=torch.float32)

    flex_attention(q, k, v, alibi_slopes=slopes.clone())
    explicit_passed = capture_backend["kwargs"]["alibi_slopes"].clone()

    def alibi_score_mod(score, b, h, qi, ki):
        slope = 2.0 ** (-8.0 * float(h) / H)
        return score + slope * (ki - qi)

    capture_backend.clear()
    flex_attention(q, k, v, score_mod=alibi_score_mod)
    detected_passed = capture_backend["kwargs"]["alibi_slopes"].clone()

    assert torch.allclose(explicit_passed.cpu(), detected_passed.cpu(), atol=1e-5)


def test_explicit_alibi_with_causal_mask(capture_backend):
    # ALiBi is commonly paired with causal; the explicit slopes must still flow.
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    bm = _DummyBlockMask(lambda b, h, qi, ki: qi >= ki)
    flex_attention(q, k, v, alibi_slopes=slopes.clone(), block_mask=bm)
    assert capture_backend["kwargs"]["causal"] is True
    assert torch.allclose(capture_backend["kwargs"]["alibi_slopes"].cpu(), slopes)


def test_explicit_alibi_with_nontrivial_score_mod_raises(capture_backend):
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)

    def score_mod(score, b, h, qi, ki):
        return score + 0.1 * (ki - qi)

    with pytest.raises(ValueError):
        flex_attention(q, k, v, alibi_slopes=slopes, score_mod=score_mod)


def test_explicit_alibi_with_identity_score_mod_allowed(capture_backend):
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)

    def identity(score, b, h, qi, ki):
        return score

    flex_attention(q, k, v, alibi_slopes=slopes.clone(), score_mod=identity)
    assert torch.allclose(capture_backend["kwargs"]["alibi_slopes"].cpu(), slopes)


def test_explicit_alibi_invalid_length_raises_via_entry(capture_backend):
    q, k, v = _make_qkv(Hq=8)
    slopes = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)  # len 3 != Hq=8
    with pytest.raises(ValueError):
        flex_attention(q, k, v, alibi_slopes=slopes)


def test_explicit_softcap_positive_raises_not_implemented(capture_backend):
    q, k, v = _make_qkv()
    with pytest.raises(NotImplementedError):
        flex_attention(q, k, v, softcap=30.0)
    # The gate fires before any backend dispatch: the cap is never silently dropped.
    assert "called" not in capture_backend


def test_explicit_softcap_zero_and_none_are_noops(capture_backend):
    q, k, v = _make_qkv()
    out = flex_attention(q, k, v, softcap=0.0)
    assert out.shape == (1, 4, 16, 16)
    assert capture_backend["kwargs"]["alibi_slopes"] is None

    capture_backend.clear()
    flex_attention(q, k, v, softcap=None)
    assert capture_backend["called"] is True


def test_explicit_softcap_positive_raises_even_with_explicit_alibi(capture_backend):
    # softcap>0 is the blocker; it must still hard-error alongside explicit alibi.
    H = 4
    q, k, v = _make_qkv(Hq=H)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        flex_attention(q, k, v, alibi_slopes=slopes, softcap=15.0)


def test_no_explicit_args_matches_plain_dispatch(capture_backend):
    # Zero-regression guard: without the extension args the turbo path is taken
    # with alibi_slopes=None, exactly as before.
    q, k, v = _make_qkv()
    out = flex_attention(q, k, v)
    assert out.shape == (1, 4, 16, 16)
    assert capture_backend["kwargs"]["alibi_slopes"] is None
    assert capture_backend["kwargs"]["causal"] is False


# ---------------------------------------------------------------------------
# Turbo-extension passthrough args: dropout_p validation helper
# ---------------------------------------------------------------------------


def test_validate_dropout_p_zero_and_valid_values():
    assert _validate_dropout_p(0.0) == 0.0
    assert _validate_dropout_p(0) == 0.0
    assert abs(_validate_dropout_p(0.1) - 0.1) < 1e-9
    assert abs(_validate_dropout_p(0.999) - 0.999) < 1e-9


def test_validate_dropout_p_rejects_one_and_above():
    with pytest.raises(ValueError):
        _validate_dropout_p(1.0)
    with pytest.raises(ValueError):
        _validate_dropout_p(1.5)


def test_validate_dropout_p_rejects_negative():
    with pytest.raises(ValueError):
        _validate_dropout_p(-0.1)


def test_validate_dropout_p_rejects_nan():
    with pytest.raises(ValueError):
        _validate_dropout_p(float("nan"))


def test_validate_dropout_p_rejects_non_number():
    with pytest.raises(ValueError):
        _validate_dropout_p([0.1])


# ---------------------------------------------------------------------------
# Turbo-extension passthrough args: sink validation helper
# ---------------------------------------------------------------------------


def test_validate_explicit_sink_ok():
    sink = torch.zeros(4, dtype=torch.float32)
    out = _validate_explicit_sink(sink, hq=4, head_dim_qk=64, head_dim_v=64, device=torch.device("cpu"))
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_validate_explicit_sink_rejects_non_tensor():
    with pytest.raises(ValueError):
        _validate_explicit_sink([0.0] * 4, hq=4, head_dim_qk=64, head_dim_v=64, device=torch.device("cpu"))


def test_validate_explicit_sink_rejects_2d():
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros((2, 4), dtype=torch.float32),
            hq=4,
            head_dim_qk=64,
            head_dim_v=64,
            device=torch.device("cpu"),
        )


def test_validate_explicit_sink_rejects_wrong_length():
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(3, dtype=torch.float32),
            hq=8,
            head_dim_qk=64,
            head_dim_v=64,
            device=torch.device("cpu"),
        )


def test_validate_explicit_sink_rejects_non_fp32():
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float16),
            hq=4,
            head_dim_qk=64,
            head_dim_v=64,
            device=torch.device("cpu"),
        )


def test_validate_explicit_sink_rejects_mismatched_head_dim():
    # Sink kernel path requires head_dim_qk == head_dim_v.
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float32),
            hq=4,
            head_dim_qk=128,
            head_dim_v=64,
            device=torch.device("cpu"),
        )


def test_validate_explicit_sink_rejects_non_pow2_head_dim():
    # Sink kernel path requires a power-of-two head dim (48 is not).
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float32),
            hq=4,
            head_dim_qk=48,
            head_dim_v=48,
            device=torch.device("cpu"),
        )


# ---------------------------------------------------------------------------
# Turbo-extension passthrough args: end-to-end dispatch (backend mocked on CPU)
# ---------------------------------------------------------------------------


def test_dropout_p_default_zero_and_sink_none_passthrough(capture_backend):
    # Zero-regression: defaults thread dropout_p=0.0 and sink=None to the backend.
    q, k, v = _make_qkv()
    out = flex_attention(q, k, v)
    assert out.shape == (1, 4, 16, 16)
    assert capture_backend["kwargs"]["dropout_p"] == 0.0
    assert capture_backend["kwargs"]["sink"] is None


def test_dropout_p_positive_passed_through(capture_backend):
    q, k, v = _make_qkv()
    flex_attention(q, k, v, dropout_p=0.1)
    assert abs(capture_backend["kwargs"]["dropout_p"] - 0.1) < 1e-9


def test_dropout_p_out_of_range_raises_via_entry(capture_backend):
    q, k, v = _make_qkv()
    with pytest.raises(ValueError):
        flex_attention(q, k, v, dropout_p=1.0)
    # The validation fires before any backend dispatch.
    assert "called" not in capture_backend


def test_sink_passed_through(capture_backend):
    # _make_qkv default D=16 (power of two), Hq=4 -> a valid len-4 fp32 sink.
    q, k, v = _make_qkv(Hq=4, D=16)
    sink = torch.arange(4, dtype=torch.float32)
    flex_attention(q, k, v, sink=sink.clone())
    passed = capture_backend["kwargs"]["sink"]
    assert passed is not None
    assert passed.shape == (4,)
    assert passed.dtype == torch.float32
    assert torch.allclose(passed.cpu(), sink)


def test_sink_invalid_length_raises_via_entry(capture_backend):
    q, k, v = _make_qkv(Hq=4, D=16)
    sink = torch.zeros(3, dtype=torch.float32)  # len 3 != Hq=4
    with pytest.raises(ValueError):
        flex_attention(q, k, v, sink=sink)
    assert "called" not in capture_backend


def test_sink_non_pow2_head_dim_raises_via_entry(capture_backend):
    # D=48 is not a power of two -> the sink kernel-path constraint rejects it.
    q, k, v = _make_qkv(Hq=4, D=48)
    sink = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError):
        flex_attention(q, k, v, sink=sink)
    assert "called" not in capture_backend


def test_dropout_and_sink_default_off_matches_plain_dispatch(capture_backend):
    # Full zero-regression guard for the two new args together.
    q, k, v = _make_qkv()
    flex_attention(q, k, v)
    assert capture_backend["kwargs"]["dropout_p"] == 0.0
    assert capture_backend["kwargs"]["sink"] is None
    assert capture_backend["kwargs"]["alibi_slopes"] is None
    assert capture_backend["kwargs"]["bias"] is None


# ---------------------------------------------------------------------------
# Turbo-extension passthrough args: bias validation / adaptation helper
# ---------------------------------------------------------------------------


def test_validate_and_adapt_bias_2d_ok():
    bias = torch.randn(16, 16, dtype=torch.float32)
    out = _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert out.shape == (16, 16)
    assert out.dtype == torch.bfloat16  # adapted to q's dtype
    assert out.is_contiguous()


def test_validate_and_adapt_bias_leading_singletons_squeezed():
    for shape in ((1, 16, 16), (1, 1, 16, 16)):
        bias = torch.randn(*shape, dtype=torch.float16)
        out = _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.float16, device=torch.device("cpu"))
        assert out.shape == (16, 16)


def test_validate_and_adapt_bias_rectangular_ok():
    bias = torch.randn(8, 16, dtype=torch.bfloat16)
    out = _validate_and_adapt_bias(bias, sq=8, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))
    assert out.shape == (8, 16)


def test_validate_and_adapt_bias_rejects_per_head_4d():
    # A genuine per-head bias cannot map to the kernel's single [Sq,Skv] bias.
    bias = torch.randn(2, 4, 16, 16, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


def test_validate_and_adapt_bias_rejects_per_batch_3d():
    bias = torch.randn(2, 16, 16, dtype=torch.bfloat16)  # leading dim 2 != 1
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


def test_validate_and_adapt_bias_rejects_wrong_last_dims():
    bias = torch.randn(16, 8, dtype=torch.bfloat16)  # skv 8 != 16
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


def test_validate_and_adapt_bias_rejects_non_tensor():
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(
            [[0.0] * 16] * 16, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu")
        )


def test_validate_and_adapt_bias_rejects_non_float():
    bias = torch.zeros(16, 16, dtype=torch.int32)
    with pytest.raises(ValueError):
        _validate_and_adapt_bias(bias, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# Turbo-extension passthrough args: bias end-to-end dispatch (backend mocked)
# ---------------------------------------------------------------------------


def test_bias_passed_through_adapted(capture_backend):
    # A 2D [Sq,Skv] fp32 bias is adapted to q's dtype and threaded to the backend.
    q, k, v = _make_qkv(Hq=4, S=16, D=16, dtype=torch.bfloat16)
    bias = torch.randn(16, 16, dtype=torch.float32)
    flex_attention(q, k, v, bias=bias)
    passed = capture_backend["kwargs"]["bias"]
    assert passed is not None
    assert passed.shape == (16, 16)
    assert passed.dtype == torch.bfloat16  # adapted to q's dtype


def test_bias_default_none_passthrough(capture_backend):
    q, k, v = _make_qkv()
    flex_attention(q, k, v)
    assert capture_backend["kwargs"]["bias"] is None


def test_bias_per_head_raises_via_entry(capture_backend):
    q, k, v = _make_qkv(Hq=4, S=16, D=16)
    bias = torch.randn(1, 4, 16, 16, dtype=torch.float16)  # per-head (H=4) not supported
    with pytest.raises(ValueError):
        flex_attention(q, k, v, bias=bias)
    assert "called" not in capture_backend


def test_bias_leading_singleton_4d_passed_through(capture_backend):
    # [1,1,Sq,Skv] is accepted (shared across batch/head) and squeezed to [Sq,Skv].
    q, k, v = _make_qkv(Hq=4, S=16, D=16, dtype=torch.float16)
    bias = torch.randn(1, 1, 16, 16, dtype=torch.float16)
    flex_attention(q, k, v, bias=bias)
    passed = capture_backend["kwargs"]["bias"]
    assert passed is not None
    assert passed.shape == (16, 16)
    assert passed.dtype == torch.float16


# ===========================================================================
# Varlen / document-packing entry (flex_attention_varlen)
#
# Pure-logic coverage of the cu_seqlens / qkv / window / max_seqlen validators
# and the end-to-end parameter-passing path (backend mocked on CPU, THD layout).
# ===========================================================================


def _cu_from_seqlens(seqlens, device="cpu"):
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)
    return cu, max(seqlens), int(cu[-1].item())


def _make_thd(total, H, D, dtype=torch.float16):
    return torch.randn(total, H, D, dtype=dtype)


# ---- _validate_max_seqlen -------------------------------------------------


def test_validate_max_seqlen_ok():
    assert _validate_max_seqlen("max_seqlen_q", 256) == 256


def test_validate_max_seqlen_rejects_float():
    with pytest.raises(ValueError):
        _validate_max_seqlen("max_seqlen_q", 256.0)


def test_validate_max_seqlen_rejects_bool():
    with pytest.raises(ValueError):
        _validate_max_seqlen("max_seqlen_q", True)


def test_validate_max_seqlen_rejects_non_positive():
    with pytest.raises(ValueError):
        _validate_max_seqlen("max_seqlen_q", 0)
    with pytest.raises(ValueError):
        _validate_max_seqlen("max_seqlen_q", -5)


# ---- _validate_window_size ------------------------------------------------


def test_validate_window_size_full_and_left_window():
    assert _validate_window_size((-1, -1)) == (-1, -1)
    assert _validate_window_size((256, 0)) == (256, 0)
    assert _validate_window_size([128, 0]) == (128, 0)  # list accepted, coerced to tuple


def test_validate_window_size_rejects_wrong_length():
    with pytest.raises(ValueError):
        _validate_window_size((1, 2, 3))


def test_validate_window_size_rejects_non_int():
    with pytest.raises(ValueError):
        _validate_window_size((128.0, 0))
    with pytest.raises(ValueError):
        _validate_window_size((True, 0))  # bool is not accepted as a window bound


def test_validate_window_size_rejects_non_sequence():
    with pytest.raises(ValueError):
        _validate_window_size(128)
    with pytest.raises(ValueError):
        _validate_window_size(torch.tensor([128, 0]))


# ---- _validate_qkv_varlen -------------------------------------------------


def test_validate_qkv_varlen_ok_mha():
    q = _make_thd(512, 8, 128)
    _validate_qkv_varlen(q, q.clone(), q.clone())  # no raise


def test_validate_qkv_varlen_ok_gqa():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 2, 128)
    _validate_qkv_varlen(q, k, k.clone())  # Hq=8, Hkv=2 -> ok


def test_validate_qkv_varlen_rejects_4d():
    q = torch.randn(1, 512, 8, 128, dtype=torch.float16)
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, q.clone(), q.clone())


def test_validate_qkv_varlen_rejects_fp32():
    q = _make_thd(512, 8, 128, dtype=torch.float32)
    with pytest.raises(NotImplementedError):
        _validate_qkv_varlen(q, q.clone(), q.clone())


def test_validate_qkv_varlen_rejects_kv_total_mismatch():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 8, 128)
    v = _make_thd(256, 8, 128)  # total_v != total_k
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, k, v)


def test_validate_qkv_varlen_rejects_kv_head_mismatch():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 4, 128)
    v = _make_thd(512, 8, 128)  # Hv != Hk
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, k, v)


def test_validate_qkv_varlen_rejects_head_dim_qk_mismatch():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 8, 64)  # Dk != Dq
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, k, k.clone())


def test_validate_qkv_varlen_rejects_non_divisible_heads():
    q = _make_thd(512, 8, 128)
    k = _make_thd(512, 3, 128)  # 8 % 3 != 0
    with pytest.raises(ValueError):
        _validate_qkv_varlen(q, k, k.clone())


# ---- _validate_cu_seqlens -------------------------------------------------


def test_validate_cu_seqlens_ok_causal():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    got = _validate_cu_seqlens(
        cu,
        cu,
        total_q=total,
        total_k=total,
        max_seqlen_q=max_s,
        max_seqlen_k=max_s,
        causal=True,
        device=torch.device("cpu"),
    )
    assert got == (256, 256)


def test_validate_cu_seqlens_ok_full_cross_lengths():
    # Non-causal cross attention: q and k may have different per-segment lengths
    # (same number of segments), which is allowed when causal=False.
    cu_q, max_q, total_q = _cu_from_seqlens([128, 256])
    cu_k, max_k, total_k = _cu_from_seqlens([300, 84])
    _validate_cu_seqlens(
        cu_q,
        cu_k,
        total_q=total_q,
        total_k=total_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        causal=False,
        device=torch.device("cpu"),
    )  # no raise


def test_validate_cu_seqlens_rejects_non_int32():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    cu_long = cu.to(torch.int64)
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu_long,
            cu_long,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_non_1d():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    cu2d = cu.view(1, -1)
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu2d,
            cu2d,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_too_short():
    cu = torch.zeros(1, dtype=torch.int32)  # numel < 2
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu,
            cu,
            total_q=0,
            total_k=0,
            max_seqlen_q=1,
            max_seqlen_k=1,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_nonzero_first():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    bad = cu.clone()
    bad[0] = 5  # first must be 0
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            bad,
            bad,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_non_monotone():
    bad = torch.tensor([0, 256, 128, 512], dtype=torch.int32)  # decreasing in the middle
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            bad,
            bad,
            total_q=512,
            total_k=512,
            max_seqlen_q=256,
            max_seqlen_k=256,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_last_ne_total():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu,
            cu,
            total_q=total + 1,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_length_mismatch():
    cu_q, max_q, total = _cu_from_seqlens([128, 128, 256])  # len 4
    cu_k, max_k, _ = _cu_from_seqlens([256, 256])  # len 3, same total 512
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu_q,
            cu_k,
            total_q=total,
            total_k=512,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=False,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_max_seqlen_too_small():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu,
            cu,
            total_q=total,
            total_k=total,
            max_seqlen_q=100,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_causal_len_mismatch():
    cu_q, max_q, total = _cu_from_seqlens([128, 128, 256])
    cu_k, max_k, _ = _cu_from_seqlens([256, 128, 128])  # same total/segments, different split
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu_q,
            cu_k,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            causal=True,
            device=torch.device("cpu"),
        )


def test_validate_cu_seqlens_rejects_device_mismatch():
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])  # on cpu
    with pytest.raises(ValueError):
        _validate_cu_seqlens(
            cu,
            cu,
            total_q=total,
            total_k=total,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            causal=True,
            device=torch.device("meta"),  # cpu != meta
        )


# ---- end-to-end dispatch (backend mocked on CPU) --------------------------


@pytest.fixture
def capture_varlen_backend(monkeypatch):
    captured = {}

    def fake_flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
    ):
        captured["called"] = True
        captured["kwargs"] = kwargs
        captured["cu_q"] = cu_seqlens_q
        captured["cu_k"] = cu_seqlens_k
        captured["max_q"] = max_seqlen_q
        captured["max_k"] = max_seqlen_k
        captured["q_shape"] = tuple(q.shape)
        captured["k_shape"] = tuple(k.shape)
        captured["v_shape"] = tuple(v.shape)
        if kwargs.get("return_lse"):
            return q.clone(), torch.zeros((q.shape[1], q.shape[0]), dtype=torch.float32)
        return q.clone()

    monkeypatch.setattr(
        "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_varlen_func",
        fake_flash_attn_varlen_func,
        raising=True,
    )
    return captured


def test_varlen_causal_thd_passthrough(capture_varlen_backend):
    # THD is threaded to the backend verbatim (no transpose), with causal=True and
    # the defaults for every optional arg.
    H, D = 8, 128
    q = _make_thd(512, H, D)
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    out = flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True)
    assert capture_varlen_backend["q_shape"] == (512, H, D)  # unchanged: no transpose
    assert out.shape == (512, H, D)
    kw = capture_varlen_backend["kwargs"]
    assert kw["causal"] is True
    assert kw["window_size"] == (-1, -1)
    assert kw["dropout_p"] == 0.0
    assert kw["alibi_slopes"] is None
    # sink is threaded only when supplied (newer-backend feature) -> absent here.
    assert kw.get("sink") is None
    # bias is not exposed by the varlen entry -> left to the backend default (absent).
    assert kw.get("bias") is None
    assert kw["deterministic"] is False
    assert capture_varlen_backend["cu_q"] is cu  # passed through unmodified
    assert capture_varlen_backend["max_q"] == max_s


def test_varlen_window_passed_through(capture_varlen_backend):
    q = _make_thd(512, 8, 128)
    cu, max_s, total = _cu_from_seqlens([256, 256])
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, window_size=(128, 0))
    assert capture_varlen_backend["kwargs"]["window_size"] == (128, 0)


def test_varlen_scale_passed_through(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, scale=0.5)
    assert capture_varlen_backend["kwargs"]["softmax_scale"] == 0.5


def test_varlen_dropout_passed_through(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, dropout_p=0.1)
    assert abs(capture_varlen_backend["kwargs"]["dropout_p"] - 0.1) < 1e-9


def test_varlen_dropout_out_of_range_raises(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, dropout_p=1.0)
    assert "called" not in capture_varlen_backend


def test_varlen_alibi_passed_through(capture_varlen_backend):
    H, D = 4, 128
    q = _make_thd(256, H, D)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    flex_attention_varlen(
        q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, alibi_slopes=slopes.clone()
    )
    passed = capture_varlen_backend["kwargs"]["alibi_slopes"]
    assert passed is not None
    assert passed.dtype == torch.float32
    assert torch.allclose(passed.cpu(), slopes)


def test_varlen_alibi_invalid_length_raises(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    slopes = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)  # len 3 != Hq=8
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, alibi_slopes=slopes)
    assert "called" not in capture_varlen_backend


def test_varlen_sink_passed_through(capture_varlen_backend):
    H, D = 4, 128  # D power of two, Hq=4
    q = _make_thd(256, H, D)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    sink = torch.arange(H, dtype=torch.float32)
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, sink=sink.clone())
    passed = capture_varlen_backend["kwargs"]["sink"]
    assert passed is not None
    assert passed.shape == (H,)
    assert torch.allclose(passed.cpu(), sink)


def test_varlen_sink_invalid_length_raises(capture_varlen_backend):
    q = _make_thd(256, 4, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    sink = torch.zeros(3, dtype=torch.float32)  # len 3 != Hq=4
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, sink=sink)
    assert "called" not in capture_varlen_backend


def test_varlen_softcap_positive_raises_not_implemented(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    with pytest.raises(NotImplementedError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, softcap=30.0)
    # The gate fires before any backend dispatch: the cap is never silently dropped.
    assert "called" not in capture_varlen_backend


def test_varlen_softcap_zero_and_none_are_noops(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, softcap=0.0)
    assert capture_varlen_backend["called"] is True
    capture_varlen_backend.clear()
    flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, softcap=None)
    assert capture_varlen_backend["called"] is True


def test_varlen_invalid_cu_raises_before_dispatch(capture_varlen_backend):
    q = _make_thd(512, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128, 256])
    bad = cu.to(torch.int64)  # wrong dtype
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), bad, bad, max_s, max_s, causal=True)
    assert "called" not in capture_varlen_backend


def test_varlen_cu_total_mismatch_raises(capture_varlen_backend):
    # q has 512 tokens but cu_seqlens says 384 -> rejected before dispatch.
    q = _make_thd(512, 8, 128)
    cu, max_s, _ = _cu_from_seqlens([128, 256])  # total 384 != 512
    with pytest.raises(ValueError):
        flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True)
    assert "called" not in capture_varlen_backend


def test_varlen_return_lse_returns_tuple(capture_varlen_backend):
    q = _make_thd(256, 8, 128)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    out = flex_attention_varlen(q, q.clone(), q.clone(), cu, cu, max_s, max_s, causal=True, return_lse=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert capture_varlen_backend["kwargs"]["return_lse"] is True


def test_varlen_gqa_passthrough(capture_varlen_backend):
    Hq, Hkv, D = 8, 2, 128
    q = _make_thd(256, Hq, D)
    k = _make_thd(256, Hkv, D)
    cu, max_s, total = _cu_from_seqlens([128, 128])
    flex_attention_varlen(q, k, k.clone(), cu, cu, max_s, max_s, causal=True)
    assert capture_varlen_backend["q_shape"] == (256, Hq, D)
    assert capture_varlen_backend["k_shape"] == (256, Hkv, D)


def test_varlen_full_cross_attention_dispatches(capture_varlen_backend):
    # causal=False with different q/k per-segment lengths (cross attention) dispatches.
    q = _make_thd(384, 8, 128)
    kv = _make_thd(512, 8, 128)
    cu_q, max_q, total_q = _cu_from_seqlens([128, 256])
    cu_k, max_k, total_k = _cu_from_seqlens([256, 256])
    flex_attention_varlen(q, kv, kv.clone(), cu_q, cu_k, max_q, max_k, causal=False)
    kw = capture_varlen_backend["kwargs"]
    assert kw["causal"] is False
    assert capture_varlen_backend["q_shape"] == (384, 8, 128)
    assert capture_varlen_backend["k_shape"] == (512, 8, 128)


# ===========================================================================
# Document-causal recognition on the dense entry (block_mask -> varlen routing)
# ===========================================================================


def _doc_causal_dense_mask(seg_lens):
    total = sum(seg_lens)
    document_id = torch.cat([torch.full((s,), i, dtype=torch.int64) for i, s in enumerate(seg_lens)])
    qi = torch.arange(total).view(total, 1)
    ki = torch.arange(total).view(1, total)
    return (document_id.view(total, 1) == document_id.view(1, total)) & (qi >= ki)


def _doc_causal_block_mask(seg_lens):
    document_id = torch.cat([torch.full((s,), i, dtype=torch.int64) for i, s in enumerate(seg_lens)])

    def mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        return same_doc & (q_idx >= kv_idx)

    return _DummyBlockMask(mask_mod)


# ---- _detect_document_causal_segments -------------------------------------


def test_detect_document_segments_basic():
    seg = [128, 128, 256]
    mask = _doc_causal_dense_mask(seg)
    got = _detect_document_causal_segments(mask, q_len=512, kv_len=512, q_probe=512, kv_probe=512)
    assert got == seg


def test_detect_document_segments_two_docs():
    seg = [200, 312]
    mask = _doc_causal_dense_mask(seg)
    got = _detect_document_causal_segments(mask, q_len=512, kv_len=512, q_probe=512, kv_probe=512)
    assert got == seg


def test_detect_document_segments_none_for_plain_causal():
    n = 64
    qi = torch.arange(n).view(n, 1)
    ki = torch.arange(n).view(1, n)
    mask = qi >= ki
    assert _detect_document_causal_segments(mask, q_len=n, kv_len=n, q_probe=n, kv_probe=n) is None


def test_detect_document_segments_none_for_swa():
    n, W = 64, 8
    qi = torch.arange(n).view(n, 1)
    ki = torch.arange(n).view(1, n)
    mask = (qi >= ki) & ((qi - ki) <= W)
    assert _detect_document_causal_segments(mask, q_len=n, kv_len=n, q_probe=n, kv_probe=n) is None


def test_detect_document_segments_none_when_truncated():
    # Probe smaller than the sequence -> boundaries beyond the probe are unknowable.
    seg = [128, 128]
    mask = _doc_causal_dense_mask(seg)[:128, :128]
    assert _detect_document_causal_segments(mask, q_len=256, kv_len=256, q_probe=128, kv_probe=128) is None


def test_detect_document_segments_none_for_non_square():
    seg = [128, 128]
    mask = _doc_causal_dense_mask(seg)
    assert _detect_document_causal_segments(mask, q_len=256, kv_len=200, q_probe=256, kv_probe=200) is None


def test_detect_document_segments_none_for_doc_with_hole():
    # Block diagonal + causal but with one interior position masked out -> not exact.
    seg = [128, 128]
    mask = _doc_causal_dense_mask(seg).clone()
    mask[10, 5] = False  # a hole inside doc 0's causal region
    assert _detect_document_causal_segments(mask, q_len=256, kv_len=256, q_probe=256, kv_probe=256) is None


# ---- classify returns document_causal -------------------------------------


def test_classify_document_causal():
    bm = _doc_causal_block_mask([128, 128, 256])
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=512, kv_len=512)
    assert cfg["kind"] == "document_causal"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (-1, -1)
    assert cfg["doc_seglens"] == [128, 128, 256]


def test_classify_document_single_doc_is_causal():
    # A single document is just plain causal, not the document-packing kind.
    bm = _doc_causal_block_mask([256])
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=256, kv_len=256)
    assert cfg["kind"] == "causal"


# ---- end-to-end: dense entry routes a document mask to the varlen backend --


def _make_bhsd(B, H, S, D, dtype=torch.float16):
    return torch.randn(B, H, S, D, dtype=dtype)


def test_flex_document_routes_to_varlen(capture_varlen_backend):
    seg = [128, 128, 256]
    total, H, D = 512, 8, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    out = flex_attention(q, q.clone(), q.clone(), block_mask=bm)
    assert capture_varlen_backend["called"] is True
    # bhsd [1,H,S,D] packed to THD [B*S, H, D].
    assert capture_varlen_backend["q_shape"] == (total, H, D)
    kw = capture_varlen_backend["kwargs"]
    assert kw["causal"] is True
    assert capture_varlen_backend["cu_q"].tolist() == [0, 128, 256, 512]
    assert capture_varlen_backend["max_q"] == 256
    # output unpacked back to bhsd
    assert out.shape == (1, H, total, D)


def test_flex_document_b2_cu_replicated(capture_varlen_backend):
    # Batch > 1 with identical (batch-independent) doc structure replicates cu_seqlens.
    seg = [128, 128]
    total, B, H, D = 256, 2, 8, 128
    q = _make_bhsd(B, H, total, D)
    bm = _doc_causal_block_mask(seg)
    out = flex_attention(q, q.clone(), q.clone(), block_mask=bm)
    assert capture_varlen_backend["cu_q"].tolist() == [0, 128, 256, 384, 512]
    assert capture_varlen_backend["q_shape"] == (B * total, H, D)
    assert out.shape == (B, H, total, D)


def test_flex_document_with_explicit_alibi(capture_varlen_backend):
    seg = [128, 128]
    total, H, D = 256, 4, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    flex_attention(q, q.clone(), q.clone(), block_mask=bm, alibi_slopes=slopes.clone())
    passed = capture_varlen_backend["kwargs"]["alibi_slopes"]
    assert passed is not None
    assert torch.allclose(passed.cpu(), slopes)
    assert capture_varlen_backend["kwargs"]["causal"] is True


def test_flex_document_bias_rejected(capture_varlen_backend):
    seg = [128, 128]
    total, H, D = 256, 8, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    bias = torch.randn(total, total, dtype=torch.float16)
    with pytest.raises(NotImplementedError):
        flex_attention(q, q.clone(), q.clone(), block_mask=bm, bias=bias)
    assert "called" not in capture_varlen_backend


def test_flex_document_return_lse_rejected(capture_varlen_backend):
    seg = [128, 128]
    total, H, D = 256, 8, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    with pytest.raises(NotImplementedError):
        flex_attention(q, q.clone(), q.clone(), block_mask=bm, return_lse=True)
    assert "called" not in capture_varlen_backend


# ===========================================================================
# Document packing beyond the 512 probe grid (_locate_document_segments)
# ===========================================================================


def test_classify_document_beyond_probe_short_docs():
    # Docs shorter than the probe grid, sequence longer than it: the probed corner is
    # block-diagonal (neither causal nor a window), so recovery goes through the
    # full-sequence locator.
    seg = [128] * 8  # total 1024 > 512
    bm = _doc_causal_block_mask(seg)
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=1024, kv_len=1024)
    assert cfg["kind"] == "document_causal"
    assert cfg["doc_seglens"] == seg


def test_classify_document_beyond_probe_first_doc_longer_than_probe():
    # The first document covers the whole probe grid, so the corner looks exactly
    # causal; only the far-position check plus the locator can tell it apart from a
    # plain causal / large sliding window mask.
    seg = [1024, 512, 512]
    bm = _doc_causal_block_mask(seg)
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=2048, kv_len=2048)
    assert cfg["kind"] == "document_causal"
    assert cfg["doc_seglens"] == seg


def test_classify_document_beyond_probe_uneven_segments():
    seg = [37, 611, 1, 200, 175]
    total = sum(seg)
    bm = _doc_causal_block_mask(seg)
    cfg = _classify_block_mask(bm, B=1, H=1, q_len=total, kv_len=total)
    assert cfg["kind"] == "document_causal"
    assert cfg["doc_seglens"] == seg


def test_locate_document_segments_direct():
    seg = [300, 300, 424]
    bm = _doc_causal_block_mask(seg)
    assert fa_mod._locate_document_segments(bm.mask_mod, q_len=1024, kv_len=1024) == seg


def test_locate_document_segments_single_doc_returns_none():
    # No boundary at all == plain causal; the locator must not claim a document mask.
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    assert fa_mod._locate_document_segments(causal, q_len=1024, kv_len=1024) is None


def test_locate_document_segments_rejects_holed_mask():
    # Block-diagonal boundaries look right on the sub-diagonal, but an extra hole makes
    # the exact reconstruction fail -> None (caller raises, never silently wrong).
    seg = [256, 256, 512]
    base = _doc_causal_block_mask(seg).mask_mod

    def holed(b, h, q_idx, kv_idx):
        return base(b, h, q_idx, kv_idx) & ~((q_idx == 700) & (kv_idx == 600))

    assert fa_mod._locate_document_segments(holed, q_len=1024, kv_len=1024) is None


def test_locate_document_segments_rejects_window_mask():
    # A large sliding window also has an invisible far corner; it must not be mistaken
    # for document packing.
    def swa(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= 600)

    assert fa_mod._locate_document_segments(swa, q_len=1024, kv_len=1024) is None


def test_locate_document_segments_rejects_non_square():
    seg = [128, 128]
    bm = _doc_causal_block_mask(seg)
    assert fa_mod._locate_document_segments(bm.mask_mod, q_len=256, kv_len=512) is None


def test_locate_document_segments_beyond_exact_verify_limit():
    # Past _DOC_EXACT_VERIFY_LIMIT the O(S^2) verification is refused rather than
    # downgraded to sampling: we decline to classify instead of risking a wrong route.
    seg = [512, 512]
    bm = _doc_causal_block_mask(seg)
    limit = fa_mod._DOC_EXACT_VERIFY_LIMIT
    assert fa_mod._locate_document_segments(bm.mask_mod, q_len=limit + 1, kv_len=limit + 1) is None


def test_flex_document_beyond_probe_routes_to_varlen(capture_varlen_backend):
    # Used to raise NotImplementedError (S > 512); now recovered and routed to varlen.
    seg = [128] * 8  # total 1024 > 512
    total, H, D = 1024, 8, 128
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    out = flex_attention(q, q.clone(), q.clone(), block_mask=bm)
    assert capture_varlen_backend["called"] is True
    assert capture_varlen_backend["cu_q"].tolist() == [0, 128, 256, 384, 512, 640, 768, 896, 1024]
    assert capture_varlen_backend["max_q"] == 128
    assert out.shape == (1, H, total, D)


def test_flex_document_too_long_rejected(capture_varlen_backend):
    # Beyond the exact-verification limit the boundaries cannot be verified in full ->
    # not routed, falls through to NotImplementedError (never silently wrong).
    total = fa_mod._DOC_EXACT_VERIFY_LIMIT + 512
    seg = [512] * (total // 512)
    H, D = 2, 64
    q = _make_bhsd(1, H, total, D)
    bm = _doc_causal_block_mask(seg)
    with pytest.raises(NotImplementedError):
        flex_attention(q, q.clone(), q.clone(), block_mask=bm)
    assert "called" not in capture_varlen_backend


# ===========================================================================
# Fix B: sliding-window-causal with a window LARGER than the 512 probe grid
# ===========================================================================


def _swa_block_mask(window):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window)

    return _DummyBlockMask(mask_mod)


def _swa_mask_mod(window):
    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window)

    return mask_mod


# ---- _locate_left_window direct ------------------------------------------


@pytest.mark.parametrize("window", [256, 512, 1024, 2048, 4096])
def test_locate_left_window_recovers_w(window):
    S = 8192
    got = _locate_left_window(_swa_mask_mod(window), q_len=S, kv_len=S)
    assert got == window


def test_locate_left_window_none_for_full_causal():
    # No window (far end still visible) -> not a window; caller keeps it causal.
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    assert _locate_left_window(causal, q_len=8192, kv_len=8192) is None


def test_locate_left_window_none_for_non_translation_invariant():
    # Window shrinks at the sampled midpoint row -> exact per-row check fails.
    W = 1024
    S = 8192

    def mask_mod(b, h, q_idx, kv_idx):
        # At the exact middle row the window is smaller; everywhere else it is W.
        w = torch.where(q_idx == (S // 2), torch.as_tensor(256), torch.as_tensor(W))
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= w)

    assert _locate_left_window(mask_mod, q_len=S, kv_len=S) is None


def test_locate_left_window_none_for_hole_on_last_row():
    W = 1024
    S = 8192

    def mask_mod(b, h, q_idx, kv_idx):
        base = (q_idx >= kv_idx) & ((q_idx - kv_idx) <= W)
        hole = (q_idx == (S - 1)) & (kv_idx == (S - 1 - 500))  # inside the window
        return base & (~hole)

    assert _locate_left_window(mask_mod, q_len=S, kv_len=S) is None


def test_locate_left_window_none_for_non_square():
    assert _locate_left_window(_swa_mask_mod(1024), q_len=8192, kv_len=4096) is None


# ---- classification integration (the real fix) ---------------------------


@pytest.mark.parametrize("window", [256, 512, 1024, 2048, 4096])
def test_classify_large_window_on_long_sequence(window):
    # The exact case the baseline benchmark hit: SWA(W>512) on S=8192 was raising
    # NotImplementedError; it must now classify as sliding_window_causal(W, 0).
    cfg = _classify_block_mask(_swa_block_mask(window), B=2, H=32, q_len=8192, kv_len=8192)
    assert cfg["kind"] == "sliding_window_causal"
    assert cfg["causal"] is True
    assert cfg["window_size"] == (window, 0)


def test_classify_large_window_w1024_matches_task_case():
    cfg = _classify_block_mask(_swa_block_mask(1024), B=2, H=32, q_len=2048, kv_len=2048)
    assert cfg["window_size"] == (1024, 0)


def test_classify_full_causal_still_causal_on_long_sequence():
    # No window: far end visible -> stays causal (not misdetected as a window).
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    cfg = _classify_block_mask(_DummyBlockMask(causal), B=1, H=1, q_len=8192, kv_len=8192)
    assert cfg["kind"] == "causal"
    assert cfg["window_size"] == (-1, -1)


def test_classify_window_larger_than_seq_is_causal():
    # A window >= sequence length is effectively full causal (far end visible).
    cfg = _classify_block_mask(_swa_block_mask(100000), B=1, H=1, q_len=8192, kv_len=8192)
    assert cfg["kind"] == "causal"


def test_classify_nonstandard_long_mask_still_raises():
    # Causal in the corner + invisible far corner, but NOT a clean window (window
    # depends on position at a sampled row) -> still NotImplementedError.
    S = 4096

    def mask_mod(b, h, q_idx, kv_idx):
        w = torch.where(q_idx < (S // 2), torch.as_tensor(1024), torch.as_tensor(300))
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= w)

    with pytest.raises(NotImplementedError):
        _classify_block_mask(_DummyBlockMask(mask_mod), B=1, H=1, q_len=S, kv_len=S)


# ===========================================================================
# Fix A: classification / detection caching (perf; behaviour unchanged)
# ===========================================================================


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
    orig = fa_mod._classify_block_mask_uncached

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(fa_mod, "_classify_block_mask_uncached", spy)
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
    orig = fa_mod._detect_softcap

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(fa_mod, "_detect_softcap", spy)
    clear_classification_cache()
    sm = _softcap_score_mod(30.0)
    c1 = _cached_detect_softcap(sm, B=1, Hq=8, q_len=64, kv_len=64)
    c2 = _cached_detect_softcap(sm, B=1, Hq=8, q_len=64, kv_len=64)
    assert calls["n"] == 1  # second call hit the cache
    assert c1 == c2 and c1 is not None
    assert abs(c1 - 30.0) < 1e-2 * 30.0  # behaviour unchanged


def test_cached_detect_softcap_distinct_objects_recompute(monkeypatch):
    calls = {"n": 0}
    orig = fa_mod._detect_softcap

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(fa_mod, "_detect_softcap", spy)
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


# ===========================================================================
# ALiBi sign-convention build self-check
# ===========================================================================


def _fake_alibi_backend(sign):
    """A CPU stand-in for flash_attn_func that applies ALiBi with the given sign."""

    def fake(q, k, v, causal=False, alibi_slopes=None, **kwargs):
        q_b, k_b, v_b = (t.transpose(1, 2) for t in (q, k, v))
        out = fa_mod._reference_attention_with_alibi(q_b, k_b, v_b, alibi_slopes, sign, causal=causal)
        return out.transpose(1, 2).to(q.dtype)

    return fake


@pytest.fixture
def patch_alibi_backend(monkeypatch):
    def _apply(sign):
        monkeypatch.setattr(
            "primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func",
            _fake_alibi_backend(sign),
        )

    return _apply


def test_check_alibi_sign_detects_plus(patch_alibi_backend):
    patch_alibi_backend(1.0)
    rep = fa_mod.check_alibi_sign_convention(device="cpu", dtype=torch.float32, seqlen=64, head_dim=32)
    assert rep["sign"] == 1.0
    assert rep["matches_assumption"] is True
    assert rep["plus_err"] < rep["minus_err"]


def test_check_alibi_sign_detects_minus(patch_alibi_backend):
    patch_alibi_backend(-1.0)
    rep = fa_mod.check_alibi_sign_convention(device="cpu", dtype=torch.float32, seqlen=64, head_dim=32)
    assert rep["sign"] == -1.0
    assert rep["matches_assumption"] is False


def test_assert_alibi_sign_passes_on_matching_build(patch_alibi_backend):
    patch_alibi_backend(fa_mod._ASSUMED_ALIBI_SIGN)
    rep = fa_mod.assert_alibi_sign_convention(device="cpu", dtype=torch.float32, seqlen=64, head_dim=32)
    assert rep["matches_assumption"] is True


def test_assert_alibi_sign_raises_on_flipped_build(patch_alibi_backend):
    patch_alibi_backend(-fa_mod._ASSUMED_ALIBI_SIGN)
    with pytest.raises(RuntimeError, match="ALiBi sign convention"):
        fa_mod.assert_alibi_sign_convention(device="cpu", dtype=torch.float32, seqlen=64, head_dim=32)


def test_check_alibi_sign_reports_unknown_for_unrelated_backend(monkeypatch):
    # A backend that ignores ALiBi entirely matches neither hypothesis: report None
    # rather than picking the "less wrong" sign.
    def no_alibi(q, k, v, causal=False, alibi_slopes=None, **kwargs):
        zeros = torch.zeros_like(alibi_slopes)
        q_b, k_b, v_b = (t.transpose(1, 2) for t in (q, k, v))
        out = fa_mod._reference_attention_with_alibi(q_b, k_b, v_b, zeros, 1.0, causal=causal)
        return out.transpose(1, 2).to(q.dtype)

    monkeypatch.setattr("primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func", no_alibi)
    rep = fa_mod.check_alibi_sign_convention(device="cpu", dtype=torch.float32, seqlen=64, head_dim=32)
    assert rep["sign"] is None
    assert rep["matches_assumption"] is False


# ===========================================================================
# bshd-native entry point (transpose/copy elimination)
# ===========================================================================


def _make_bshd(B=2, S=32, H=4, D=16, dtype=torch.float16):
    return torch.randn(B, S, H, D, dtype=dtype)


def test_bshd_entry_returns_bshd_and_dispatches(capture_backend):
    B, S, H, D = 2, 32, 4, 16
    q, k, v = (_make_bshd(B, S, H, D) for _ in range(3))
    out = flex_attention_bshd(q, k, v)
    assert capture_backend["called"] is True
    # Backend sees bshd, and so does the caller -- no layout round-trip.
    assert capture_backend["q_shape"] == (B, S, H, D)
    assert out.shape == (B, S, H, D)


def test_bshd_entry_passes_qkv_without_copying(monkeypatch):
    seen = {}

    def fake(q, k, v, **kwargs):
        seen["q_ptr"] = q.data_ptr()
        seen["k_ptr"] = k.data_ptr()
        seen["v_ptr"] = v.data_ptr()
        return q.clone()

    monkeypatch.setattr("primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func", fake)
    q, k, v = (_make_bshd() for _ in range(3))
    flex_attention_bshd(q, k, v)
    # A bshd-contiguous input reaches the kernel as the very same buffer.
    assert seen["q_ptr"] == q.data_ptr()
    assert seen["k_ptr"] == k.data_ptr()
    assert seen["v_ptr"] == v.data_ptr()


def test_bshd_entry_returns_backend_output_without_copying(monkeypatch):
    made = {}

    def fake(q, k, v, **kwargs):
        out = q.clone()
        made["ptr"] = out.data_ptr()
        return out

    monkeypatch.setattr("primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func", fake)
    q, k, v = (_make_bshd() for _ in range(3))
    out = flex_attention_bshd(q, k, v)
    assert out.data_ptr() == made["ptr"]


def test_bhsd_entry_still_returns_bhsd(capture_backend):
    # The torch-compatible entry is untouched: bhsd in, bhsd out.
    B, H, S, D = 2, 4, 32, 16
    q, k, v = _make_qkv(B=B, Hq=H, S=S, D=D)
    out = flex_attention(q, k, v)
    assert capture_backend["q_shape"] == (B, S, H, D)
    assert out.shape == (B, H, S, D)


def test_bshd_and_bhsd_entries_agree(monkeypatch):
    # Same numbers either way: the bshd entry is a layout change, not a semantic one.
    def fake(q, k, v, **kwargs):
        return (q.float() + k.float() + v.float()).to(q.dtype)

    monkeypatch.setattr("primus_turbo.pytorch.ops.attention.flash_attn_interface.flash_attn_func", fake)
    B, S, H, D = 2, 32, 4, 16
    q, k, v = (_make_bshd(B, S, H, D) for _ in range(3))
    out_bshd = flex_attention_bshd(q, k, v)
    out_bhsd = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    assert torch.equal(out_bshd, out_bhsd.transpose(1, 2))


def test_bshd_entry_return_lse(capture_backend):
    B, S, H, D = 1, 32, 4, 16
    q, k, v = (_make_bshd(B, S, H, D) for _ in range(3))
    out, lse = flex_attention_bshd(q, k, v, return_lse=True)
    assert out.shape == (B, S, H, D)
    assert lse.shape == (B, H, S)


def test_bshd_entry_forwards_turbo_extension_args(capture_backend):
    B, S, H, D = 1, 32, 4, 16
    q, k, v = (_make_bshd(B, S, H, D) for _ in range(3))
    slopes = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float32)
    flex_attention_bshd(q, k, v, alibi_slopes=slopes.clone(), dropout_p=0.1, scale=0.5)
    kw = capture_backend["kwargs"]
    assert torch.allclose(kw["alibi_slopes"].cpu(), slopes)
    assert kw["dropout_p"] == pytest.approx(0.1)
    assert kw["softmax_scale"] == pytest.approx(0.5)


def test_bshd_entry_validation_errors_match_bhsd():
    # Same rejections as the bhsd entry (softcap is gated, GQA needs the flag).
    q, k, v = (_make_bshd() for _ in range(3))
    with pytest.raises(NotImplementedError):
        flex_attention_bshd(q, k, v, softcap=30.0)
    q8 = _make_bshd(H=8)
    with pytest.raises(ValueError):
        flex_attention_bshd(q8, k, v)


def test_bshd_entry_document_mask_returns_bshd(capture_varlen_backend):
    seg = [128, 128]
    S, H, D = 256, 4, 64
    q = _make_bshd(1, S, H, D)
    bm = _doc_causal_block_mask(seg)
    out = flex_attention_bshd(q, q.clone(), q.clone(), block_mask=bm)
    assert capture_varlen_backend["called"] is True
    assert out.shape == (1, S, H, D)
