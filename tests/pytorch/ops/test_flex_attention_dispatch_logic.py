"""Pure-logic unit tests for the Turbo flex_attention dispatcher.

These exercise the mask classifier and the ALiBi score_mod detector without a
GPU: they only need torch on CPU.
"""

import math

import pytest
import torch

from primus_turbo.pytorch.ops.attention.flex_attention import (
    _classify_block_mask,
    _detect_alibi_slopes,
    _detect_softcap,
    _is_identity_score_mod,
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
    flex_attention,
    flex_attention_varlen,
    register_backend_override,
)


class _DummyBlockMask:
    def __init__(self, mask_mod):
        self.mask_mod = mask_mod


@pytest.fixture(autouse=True)
def _reset_backend_overrides():
    """The override registry is module-global; keep tests independent."""
    clear_backend_overrides()
    yield
    clear_backend_overrides()


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
    got = choose_backend(
        _CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False
    )
    assert got == "turbo"


def test_choose_backend_default_turbo_across_kinds():
    for cfg in (
        {"kind": "full", "causal": False, "window_size": (-1, -1)},
        {"kind": "causal", "causal": True, "window_size": (-1, -1)},
        {"kind": "sliding_window_causal", "causal": True, "window_size": (128, 0)},
    ):
        assert (
            choose_backend(cfg, shape=(2, 4, 256, 64), dtype=torch.float16, has_alibi=True)
            == "turbo"
        )


def test_register_backend_override_routes_custom():
    # An override matching this mask kind must reroute it to the custom hook.
    register_backend_override(lambda ctx: ctx["kind"] == "causal", "custom")
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)
        == "custom"
    )
    # A non-matching kind is unaffected.
    full_cfg = {"kind": "full", "causal": False, "window_size": (-1, -1)}
    assert (
        choose_backend(full_cfg, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)
        == "turbo"
    )


def test_clear_backend_overrides_restores_turbo():
    register_backend_override(lambda ctx: True, "custom")
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)
        == "custom"
    )
    clear_backend_overrides()
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)
        == "turbo"
    )


def test_backend_override_first_match_wins():
    register_backend_override(lambda ctx: ctx["kind"] == "causal", "custom")
    register_backend_override(lambda ctx: ctx["kind"] == "causal", "turbo")
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)
        == "custom"
    )


def test_backend_override_matches_on_shape_and_softcap():
    # Matchers can key off any routing-context field, e.g. a large head dim or softcap.
    register_backend_override(lambda ctx: ctx["shape"][-1] > 128, "custom")
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 256), dtype=torch.bfloat16, has_alibi=False)
        == "custom"
    )
    assert (
        choose_backend(_CAUSAL_CFG, shape=(1, 8, 512, 128), dtype=torch.bfloat16, has_alibi=False)
        == "turbo"
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
    assert not _is_identity_score_mod(
        lambda s, b, h, q, kv: s + 1.0, B=1, Hq=4, q_len=16, kv_len=16
    )


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
            torch.zeros((2, 4), dtype=torch.float32), hq=4, head_dim_qk=64, head_dim_v=64, device=torch.device("cpu")
        )


def test_validate_explicit_sink_rejects_wrong_length():
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(3, dtype=torch.float32), hq=8, head_dim_qk=64, head_dim_v=64, device=torch.device("cpu")
        )


def test_validate_explicit_sink_rejects_non_fp32():
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float16), hq=4, head_dim_qk=64, head_dim_v=64, device=torch.device("cpu")
        )


def test_validate_explicit_sink_rejects_mismatched_head_dim():
    # Sink kernel path requires head_dim_qk == head_dim_v.
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float32), hq=4, head_dim_qk=128, head_dim_v=64, device=torch.device("cpu")
        )


def test_validate_explicit_sink_rejects_non_pow2_head_dim():
    # Sink kernel path requires a power-of-two head dim (48 is not).
    with pytest.raises(ValueError):
        _validate_explicit_sink(
            torch.zeros(4, dtype=torch.float32), hq=4, head_dim_qk=48, head_dim_v=48, device=torch.device("cpu")
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
        _validate_and_adapt_bias([[0.0] * 16] * 16, sq=16, skv=16, dtype=torch.bfloat16, device=torch.device("cpu"))


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
